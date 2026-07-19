"""Separate attribution lifecycle on circuit_discovery_runs (016 R1 QA-P2).

A failed/cancelled attribution pass must not make the completed DISCOVERY
present as failed. Adds attribution_status/progress/error. Downgrade drops them.

Revision ID: e5f6a7b8c9da
Revises: c3d4e5f6a7b8
Create Date: 2026-07-19
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = 'e5f6a7b8c9da'
down_revision: Union[str, None] = 'c3d4e5f6a7b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('circuit_discovery_runs',
                  sa.Column('attribution_status', sa.String(16), nullable=True))
    op.add_column('circuit_discovery_runs',
                  sa.Column('attribution_progress', sa.Float(), nullable=True))
    op.add_column('circuit_discovery_runs',
                  sa.Column('attribution_error', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('circuit_discovery_runs', 'attribution_error')
    op.drop_column('circuit_discovery_runs', 'attribution_progress')
    op.drop_column('circuit_discovery_runs', 'attribution_status')
